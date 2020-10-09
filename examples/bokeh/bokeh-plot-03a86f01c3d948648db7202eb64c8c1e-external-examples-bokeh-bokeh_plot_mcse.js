(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("5b655cf9-3ecf-4e85-8307-834e5069ccb4");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '5b655cf9-3ecf-4e85-8307-834e5069ccb4' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"c01369e2-4464-4f4f-a30a-10397d526097":{"roots":{"references":[{"attributes":{},"id":"21748","type":"BasicTicker"},{"attributes":{"below":[{"id":"21779"}],"center":[{"id":"21782"},{"id":"21786"}],"left":[{"id":"21783"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21823"},{"id":"21825"},{"id":"21826"},{"id":"21827"},{"id":"21830"}],"title":{"id":"21832"},"toolbar":{"id":"21797"},"toolbar_location":null,"x_range":{"id":"21771"},"x_scale":{"id":"21775"},"y_range":{"id":"21773"},"y_scale":{"id":"21777"}},"id":"21770","subtype":"Figure","type":"Plot"},{"attributes":{"children":[{"id":"21860"},{"id":"21858"}]},"id":"21861","type":"Column"},{"attributes":{"text":"mu"},"id":"21832","type":"Title"},{"attributes":{"toolbars":[{"id":"21761"},{"id":"21797"}],"tools":[{"id":"21751"},{"id":"21752"},{"id":"21753"},{"id":"21754"},{"id":"21755"},{"id":"21756"},{"id":"21757"},{"id":"21758"},{"id":"21787"},{"id":"21788"},{"id":"21789"},{"id":"21790"},{"id":"21791"},{"id":"21792"},{"id":"21793"},{"id":"21794"}]},"id":"21859","type":"ProxyToolbar"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"21837"},"ticker":{"id":"21748"}},"id":"21747","type":"LinearAxis"},{"attributes":{"below":[{"id":"21743"}],"center":[{"id":"21746"},{"id":"21750"}],"left":[{"id":"21747"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21809"},{"id":"21811"},{"id":"21812"},{"id":"21813"},{"id":"21816"}],"title":{"id":"21818"},"toolbar":{"id":"21761"},"toolbar_location":null,"x_range":{"id":"21735"},"x_scale":{"id":"21739"},"y_range":{"id":"21737"},"y_scale":{"id":"21741"}},"id":"21734","subtype":"Figure","type":"Plot"},{"attributes":{"toolbar":{"id":"21859"},"toolbar_location":"above"},"id":"21860","type":"ToolbarBox"},{"attributes":{"axis":{"id":"21743"},"ticker":null},"id":"21746","type":"Grid"},{"attributes":{"data_source":{"id":"21806"},"glyph":{"id":"21807"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21808"},"selection_glyph":null,"view":{"id":"21810"}},"id":"21809","type":"GlyphRenderer"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21847"},"ticker":{"id":"21780"}},"id":"21779","type":"LinearAxis"},{"attributes":{"end":1,"start":-0.05},"id":"21773","type":"DataRange1d"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"21849"},"ticker":{"id":"21784"}},"id":"21783","type":"LinearAxis"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2148430013731262},"id":"21825","type":"Span"},{"attributes":{},"id":"21775","type":"LinearScale"},{"attributes":{},"id":"21777","type":"LinearScale"},{"attributes":{"callback":null},"id":"21794","type":"HoverTool"},{"attributes":{},"id":"21780","type":"BasicTicker"},{"attributes":{"axis":{"id":"21779"},"ticker":null},"id":"21782","type":"Grid"},{"attributes":{"axis":{"id":"21783"},"dimension":1,"ticker":null},"id":"21786","type":"Grid"},{"attributes":{},"id":"21784","type":"BasicTicker"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21808","type":"Circle"},{"attributes":{"overlay":{"id":"21795"}},"id":"21789","type":"BoxZoomTool"},{"attributes":{},"id":"21847","type":"BasicTickFormatter"},{"attributes":{},"id":"21741","type":"LinearScale"},{"attributes":{},"id":"21788","type":"PanTool"},{"attributes":{},"id":"21787","type":"ResetTool"},{"attributes":{"source":{"id":"21829"}},"id":"21831","type":"CDSView"},{"attributes":{},"id":"21793","type":"SaveTool"},{"attributes":{},"id":"21849","type":"BasicTickFormatter"},{"attributes":{},"id":"21790","type":"WheelZoomTool"},{"attributes":{},"id":"21835","type":"BasicTickFormatter"},{"attributes":{"overlay":{"id":"21796"}},"id":"21791","type":"LassoSelectTool"},{"attributes":{},"id":"21792","type":"UndoTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21787"},{"id":"21788"},{"id":"21789"},{"id":"21790"},{"id":"21791"},{"id":"21792"},{"id":"21793"},{"id":"21794"}]},"id":"21797","type":"Toolbar"},{"attributes":{},"id":"21756","type":"UndoTool"},{"attributes":{},"id":"21837","type":"BasicTickFormatter"},{"attributes":{},"id":"21744","type":"BasicTicker"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21814","type":"Dash"},{"attributes":{"callback":null},"id":"21758","type":"HoverTool"},{"attributes":{"data_source":{"id":"21815"},"glyph":{"id":"21814"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21817"}},"id":"21816","type":"GlyphRenderer"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21751"},{"id":"21752"},{"id":"21753"},{"id":"21754"},{"id":"21755"},{"id":"21756"},{"id":"21757"},{"id":"21758"}]},"id":"21761","type":"Toolbar"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21835"},"ticker":{"id":"21744"}},"id":"21743","type":"LinearAxis"},{"attributes":{"data":{"rug_x":{"__ndarray__":"jQwCEA1Gsz9bBMLb9PjIP1Qd9Zram7E/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/aF85wG2piz9oXznAbamLP2hfOcBtqYs/aF85wG2piz9oXznAbamLP2hfOcBtqYs/pI3yRkqEyT9c/+ob+nG6P4OiBeyjALU/mx4fY+a33D9wF2c1cbTRPy51Aws2htg/P07TEgOYwz9FOncTGHy5P/BQ+ANPucc/uRO6PYJJzj9CRCWTDYpmP+UvVnGFsrI/8JR71fNwcj8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"21841"},"selection_policy":{"id":"21842"}},"id":"21815","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"21747"},"dimension":1,"ticker":null},"id":"21750","type":"Grid"},{"attributes":{"overlay":{"id":"21759"}},"id":"21753","type":"BoxZoomTool"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2515582690238702},"id":"21811","type":"Span"},{"attributes":{"children":[[{"id":"21734"},0,0],[{"id":"21770"},0,1]]},"id":"21858","type":"GridBox"},{"attributes":{},"id":"21752","type":"PanTool"},{"attributes":{},"id":"21751","type":"ResetTool"},{"attributes":{"text":"tau"},"id":"21818","type":"Title"},{"attributes":{},"id":"21757","type":"SaveTool"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.1782444431478369},"id":"21812","type":"Span"},{"attributes":{},"id":"21754","type":"WheelZoomTool"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21813","type":"Span"},{"attributes":{"overlay":{"id":"21760"}},"id":"21755","type":"LassoSelectTool"},{"attributes":{"source":{"id":"21815"}},"id":"21817","type":"CDSView"},{"attributes":{},"id":"21771","type":"DataRange1d"},{"attributes":{"source":{"id":"21806"}},"id":"21810","type":"CDSView"},{"attributes":{},"id":"21851","type":"Selection"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21821","type":"Circle"},{"attributes":{"data":{"rug_x":{"__ndarray__":"fV36E1z/6j89DycBWWfXP73VlTJ7YsE/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/Oqay45Jr6D86SHRMZcflP7HThSU1z+I/VdSaCTtd6D9Hvab2ZmSwP8UA5kQ6d4M/0GULqag1oz+aI/Yi4T7rP55DEK8H/NA/qgGd6qjX1D+VqnS/h2ThP/UnuP7VN+Q/TGXHJdeQ2z8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"21853"},"selection_policy":{"id":"21854"}},"id":"21829","type":"ColumnDataSource"},{"attributes":{},"id":"21852","type":"UnionRenderers"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21795","type":"BoxAnnotation"},{"attributes":{"source":{"id":"21820"}},"id":"21824","type":"CDSView"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21828","type":"Dash"},{"attributes":{"end":1,"start":-0.05},"id":"21737","type":"DataRange1d"},{"attributes":{},"id":"21839","type":"Selection"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.15209716424958658},"id":"21826","type":"Span"},{"attributes":{"data_source":{"id":"21820"},"glyph":{"id":"21821"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21822"},"selection_glyph":null,"view":{"id":"21824"}},"id":"21823","type":"GlyphRenderer"},{"attributes":{},"id":"21840","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"21829"},"glyph":{"id":"21828"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21831"}},"id":"21830","type":"GlyphRenderer"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"bcJe+Pxh3D88Eu5SUq3NPyL+eQAns8s/mzeOVbYJ1z+YvNPsmzTWP87j+UyletE/sIr1s8Bnzz8YN/msnr7PP8gPs0h4ec8/RNGUEkZ90z8g12riYrDUP+AlTxYjYc4/UBTOvdhAzD+AFwpyJ0DOPxDSzcUXbc4/kDNyJsikyj9AY0p3Si3PP8AWdsF70MQ/4I8dAXxLyD+grhW5nZrSPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21851"},"selection_policy":{"id":"21852"}},"id":"21820","type":"ColumnDataSource"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21827","type":"Span"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21796","type":"PolyAnnotation"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21822","type":"Circle"},{"attributes":{},"id":"21853","type":"Selection"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21759","type":"BoxAnnotation"},{"attributes":{},"id":"21854","type":"UnionRenderers"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21807","type":"Circle"},{"attributes":{},"id":"21841","type":"Selection"},{"attributes":{},"id":"21842","type":"UnionRenderers"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21760","type":"PolyAnnotation"},{"attributes":{},"id":"21735","type":"DataRange1d"},{"attributes":{},"id":"21739","type":"LinearScale"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"TOY7mm62yD/upazjteTNP/DqXfy8Lcs/HHCIRHVOzT9orzckCyTOP6y8CkfbP8w/iNx/9NF5yz+IfM+LFJ/NPygi8KloJ8w/oK0q3zNVzD/YcLH58jHPP6hWLqhoMNQ/IGaMl5nu0j8AebTCVJLRP+D0hUiIM80/uNpKBpu90z/A00Fq0J3TP2iCFjW8ldY/kLBgWUSo1j8QCPgZgPnXPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21839"},"selection_policy":{"id":"21840"}},"id":"21806","type":"ColumnDataSource"}],"root_ids":["21861"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"c01369e2-4464-4f4f-a30a-10397d526097","root_ids":["21861"],"roots":{"21861":"5b655cf9-3ecf-4e85-8307-834e5069ccb4"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();