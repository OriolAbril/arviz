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
    
      
      
    
      var element = document.getElementById("8b5b5eb9-6859-4edc-9dde-3d5325e2d526");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '8b5b5eb9-6859-4edc-9dde-3d5325e2d526' but no matching script tag was found.")
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
                    
                  var docs_json = '{"7f1701f4-a6c4-4632-aa23-d05e28876b08":{"roots":{"references":[{"attributes":{},"id":"21867","type":"LinearScale"},{"attributes":{},"id":"21842","type":"PanTool"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21940"},"ticker":{"id":"21870"}},"id":"21869","type":"LinearAxis"},{"attributes":{"axis":{"id":"21869"},"ticker":null},"id":"21872","type":"Grid"},{"attributes":{},"id":"21938","type":"BasicTickFormatter"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"21938"},"ticker":{"id":"21874"}},"id":"21873","type":"LinearAxis"},{"attributes":{},"id":"21825","type":"DataRange1d"},{"attributes":{},"id":"21870","type":"BasicTicker"},{"attributes":{"callback":null},"id":"21884","type":"HoverTool"},{"attributes":{},"id":"21940","type":"BasicTickFormatter"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"21926"},"ticker":{"id":"21838"}},"id":"21837","type":"LinearAxis"},{"attributes":{},"id":"21926","type":"BasicTickFormatter"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21928"},"ticker":{"id":"21834"}},"id":"21833","type":"LinearAxis"},{"attributes":{"text":"tau"},"id":"21908","type":"Title"},{"attributes":{"axis":{"id":"21873"},"dimension":1,"ticker":null},"id":"21876","type":"Grid"},{"attributes":{},"id":"21928","type":"BasicTickFormatter"},{"attributes":{},"id":"21874","type":"BasicTicker"},{"attributes":{"toolbars":[{"id":"21851"},{"id":"21887"}],"tools":[{"id":"21841"},{"id":"21842"},{"id":"21843"},{"id":"21844"},{"id":"21845"},{"id":"21846"},{"id":"21847"},{"id":"21848"},{"id":"21877"},{"id":"21878"},{"id":"21879"},{"id":"21880"},{"id":"21881"},{"id":"21882"},{"id":"21883"},{"id":"21884"}]},"id":"21949","type":"ProxyToolbar"},{"attributes":{"overlay":{"id":"21885"}},"id":"21879","type":"BoxZoomTool"},{"attributes":{},"id":"21878","type":"PanTool"},{"attributes":{},"id":"21877","type":"ResetTool"},{"attributes":{},"id":"21883","type":"SaveTool"},{"attributes":{},"id":"21880","type":"WheelZoomTool"},{"attributes":{"overlay":{"id":"21886"}},"id":"21881","type":"LassoSelectTool"},{"attributes":{},"id":"21846","type":"UndoTool"},{"attributes":{},"id":"21882","type":"UndoTool"},{"attributes":{"data_source":{"id":"21919"},"glyph":{"id":"21918"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21921"}},"id":"21920","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"21849"}},"id":"21843","type":"BoxZoomTool"},{"attributes":{"data_source":{"id":"21910"},"glyph":{"id":"21911"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21912"},"selection_glyph":null,"view":{"id":"21914"}},"id":"21913","type":"GlyphRenderer"},{"attributes":{"callback":null},"id":"21848","type":"HoverTool"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2148430013731262},"id":"21915","type":"Span"},{"attributes":{"data":{"rug_x":{"__ndarray__":"jQwCEA1Gsz9bBMLb9PjIP1Qd9Zram7E/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/aF85wG2piz9oXznAbamLP2hfOcBtqYs/aF85wG2piz9oXznAbamLP2hfOcBtqYs/pI3yRkqEyT9c/+ob+nG6P4OiBeyjALU/mx4fY+a33D9wF2c1cbTRPy51Aws2htg/P07TEgOYwz9FOncTGHy5P/BQ+ANPucc/uRO6PYJJzj9CRCWTDYpmP+UvVnGFsrI/8JR71fNwcj8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"21934"},"selection_policy":{"id":"21933"}},"id":"21905","type":"ColumnDataSource"},{"attributes":{"source":{"id":"21919"}},"id":"21921","type":"CDSView"},{"attributes":{"source":{"id":"21910"}},"id":"21914","type":"CDSView"},{"attributes":{},"id":"21841","type":"ResetTool"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.15209716424958658},"id":"21916","type":"Span"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21898","type":"Circle"},{"attributes":{},"id":"21838","type":"BasicTicker"},{"attributes":{"text":"mu"},"id":"21922","type":"Title"},{"attributes":{},"id":"21847","type":"SaveTool"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21918","type":"Dash"},{"attributes":{},"id":"21844","type":"WheelZoomTool"},{"attributes":{},"id":"21831","type":"LinearScale"},{"attributes":{"overlay":{"id":"21850"}},"id":"21845","type":"LassoSelectTool"},{"attributes":{},"id":"21861","type":"DataRange1d"},{"attributes":{},"id":"21865","type":"LinearScale"},{"attributes":{"data_source":{"id":"21905"},"glyph":{"id":"21904"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21907"}},"id":"21906","type":"GlyphRenderer"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"bcJe+Pxh3D88Eu5SUq3NPyL+eQAns8s/mzeOVbYJ1z+YvNPsmzTWP87j+UyletE/sIr1s8Bnzz8YN/msnr7PP8gPs0h4ec8/RNGUEkZ90z8g12riYrDUP+AlTxYjYc4/UBTOvdhAzD+AFwpyJ0DOPxDSzcUXbc4/kDNyJsikyj9AY0p3Si3PP8AWdsF70MQ/4I8dAXxLyD+grhW5nZrSPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21944"},"selection_policy":{"id":"21943"}},"id":"21910","type":"ColumnDataSource"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.1782444431478369},"id":"21902","type":"Span"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21904","type":"Dash"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21911","type":"Circle"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2515582690238702},"id":"21901","type":"Span"},{"attributes":{"data_source":{"id":"21896"},"glyph":{"id":"21897"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21898"},"selection_glyph":null,"view":{"id":"21900"}},"id":"21899","type":"GlyphRenderer"},{"attributes":{},"id":"21943","type":"UnionRenderers"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21885","type":"BoxAnnotation"},{"attributes":{},"id":"21944","type":"Selection"},{"attributes":{"source":{"id":"21905"}},"id":"21907","type":"CDSView"},{"attributes":{"axis":{"id":"21837"},"dimension":1,"ticker":null},"id":"21840","type":"Grid"},{"attributes":{"source":{"id":"21896"}},"id":"21900","type":"CDSView"},{"attributes":{},"id":"21931","type":"UnionRenderers"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21903","type":"Span"},{"attributes":{},"id":"21932","type":"Selection"},{"attributes":{"end":1,"start":-0.05},"id":"21863","type":"DataRange1d"},{"attributes":{"children":[[{"id":"21824"},0,0],[{"id":"21860"},0,1]]},"id":"21948","type":"GridBox"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21849","type":"BoxAnnotation"},{"attributes":{},"id":"21945","type":"UnionRenderers"},{"attributes":{},"id":"21946","type":"Selection"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21897","type":"Circle"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21886","type":"PolyAnnotation"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"TOY7mm62yD/upazjteTNP/DqXfy8Lcs/HHCIRHVOzT9orzckCyTOP6y8CkfbP8w/iNx/9NF5yz+IfM+LFJ/NPygi8KloJ8w/oK0q3zNVzD/YcLH58jHPP6hWLqhoMNQ/IGaMl5nu0j8AebTCVJLRP+D0hUiIM80/uNpKBpu90z/A00Fq0J3TP2iCFjW8ldY/kLBgWUSo1j8QCPgZgPnXPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21932"},"selection_policy":{"id":"21931"}},"id":"21896","type":"ColumnDataSource"},{"attributes":{},"id":"21933","type":"UnionRenderers"},{"attributes":{},"id":"21934","type":"Selection"},{"attributes":{"end":1,"start":-0.05},"id":"21827","type":"DataRange1d"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21850","type":"PolyAnnotation"},{"attributes":{},"id":"21834","type":"BasicTicker"},{"attributes":{"axis":{"id":"21833"},"ticker":null},"id":"21836","type":"Grid"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21877"},{"id":"21878"},{"id":"21879"},{"id":"21880"},{"id":"21881"},{"id":"21882"},{"id":"21883"},{"id":"21884"}]},"id":"21887","type":"Toolbar"},{"attributes":{"below":[{"id":"21869"}],"center":[{"id":"21872"},{"id":"21876"}],"left":[{"id":"21873"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21913"},{"id":"21915"},{"id":"21916"},{"id":"21917"},{"id":"21920"}],"title":{"id":"21922"},"toolbar":{"id":"21887"},"toolbar_location":null,"x_range":{"id":"21861"},"x_scale":{"id":"21865"},"y_range":{"id":"21863"},"y_scale":{"id":"21867"}},"id":"21860","subtype":"Figure","type":"Plot"},{"attributes":{"children":[{"id":"21950"},{"id":"21948"}]},"id":"21951","type":"Column"},{"attributes":{"data":{"rug_x":{"__ndarray__":"fV36E1z/6j89DycBWWfXP73VlTJ7YsE/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/Oqay45Jr6D86SHRMZcflP7HThSU1z+I/VdSaCTtd6D9Hvab2ZmSwP8UA5kQ6d4M/0GULqag1oz+aI/Yi4T7rP55DEK8H/NA/qgGd6qjX1D+VqnS/h2ThP/UnuP7VN+Q/TGXHJdeQ2z8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"21946"},"selection_policy":{"id":"21945"}},"id":"21919","type":"ColumnDataSource"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21917","type":"Span"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21912","type":"Circle"},{"attributes":{"toolbar":{"id":"21949"},"toolbar_location":"above"},"id":"21950","type":"ToolbarBox"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21841"},{"id":"21842"},{"id":"21843"},{"id":"21844"},{"id":"21845"},{"id":"21846"},{"id":"21847"},{"id":"21848"}]},"id":"21851","type":"Toolbar"},{"attributes":{},"id":"21829","type":"LinearScale"},{"attributes":{"below":[{"id":"21833"}],"center":[{"id":"21836"},{"id":"21840"}],"left":[{"id":"21837"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21899"},{"id":"21901"},{"id":"21902"},{"id":"21903"},{"id":"21906"}],"title":{"id":"21908"},"toolbar":{"id":"21851"},"toolbar_location":null,"x_range":{"id":"21825"},"x_scale":{"id":"21829"},"y_range":{"id":"21827"},"y_scale":{"id":"21831"}},"id":"21824","subtype":"Figure","type":"Plot"}],"root_ids":["21951"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"7f1701f4-a6c4-4632-aa23-d05e28876b08","root_ids":["21951"],"roots":{"21951":"8b5b5eb9-6859-4edc-9dde-3d5325e2d526"}}];
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