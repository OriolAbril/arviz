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
    
      
      
    
      var element = document.getElementById("09501135-428c-4582-9e52-32ab60c285b7");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '09501135-428c-4582-9e52-32ab60c285b7' but no matching script tag was found.")
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
                    
                  var docs_json = '{"016603b1-7e9f-425b-b569-618f0bd26624":{"roots":{"references":[{"attributes":{"callback":null},"id":"21743","type":"HoverTool"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"21834"},"ticker":{"id":"21769"}},"id":"21768","type":"LinearAxis"},{"attributes":{"axis":{"id":"21768"},"dimension":1,"ticker":null},"id":"21771","type":"Grid"},{"attributes":{},"id":"21832","type":"BasicTickFormatter"},{"attributes":{"callback":null},"id":"21779","type":"HoverTool"},{"attributes":{},"id":"21820","type":"BasicTickFormatter"},{"attributes":{},"id":"21769","type":"BasicTicker"},{"attributes":{"data":{"rug_x":{"__ndarray__":"fV36E1z/6j89DycBWWfXP73VlTJ7YsE/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/Oqay45Jr6D86SHRMZcflP7HThSU1z+I/VdSaCTtd6D9Hvab2ZmSwP8UA5kQ6d4M/0GULqag1oz+aI/Yi4T7rP55DEK8H/NA/qgGd6qjX1D+VqnS/h2ThP/UnuP7VN+Q/TGXHJdeQ2z8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"21840"},"selection_policy":{"id":"21841"}},"id":"21814","type":"ColumnDataSource"},{"attributes":{"text":"tau"},"id":"21803","type":"Title"},{"attributes":{"overlay":{"id":"21780"}},"id":"21774","type":"BoxZoomTool"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"21822"},"ticker":{"id":"21733"}},"id":"21732","type":"LinearAxis"},{"attributes":{},"id":"21773","type":"PanTool"},{"attributes":{},"id":"21772","type":"ResetTool"},{"attributes":{"data":{"rug_x":{"__ndarray__":"jQwCEA1Gsz9bBMLb9PjIP1Qd9Zram7E/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/aF85wG2piz9oXznAbamLP2hfOcBtqYs/aF85wG2piz9oXznAbamLP2hfOcBtqYs/pI3yRkqEyT9c/+ob+nG6P4OiBeyjALU/mx4fY+a33D9wF2c1cbTRPy51Aws2htg/P07TEgOYwz9FOncTGHy5P/BQ+ANPucc/uRO6PYJJzj9CRCWTDYpmP+UvVnGFsrI/8JR71fNwcj8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"21828"},"selection_policy":{"id":"21829"}},"id":"21800","type":"ColumnDataSource"},{"attributes":{},"id":"21778","type":"SaveTool"},{"attributes":{},"id":"21775","type":"WheelZoomTool"},{"attributes":{"overlay":{"id":"21781"}},"id":"21776","type":"LassoSelectTool"},{"attributes":{"axis":{"id":"21728"},"ticker":null},"id":"21731","type":"Grid"},{"attributes":{},"id":"21777","type":"UndoTool"},{"attributes":{},"id":"21741","type":"UndoTool"},{"attributes":{"text":"mu"},"id":"21817","type":"Title"},{"attributes":{"source":{"id":"21805"}},"id":"21809","type":"CDSView"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.15209716424958658},"id":"21811","type":"Span"},{"attributes":{"data_source":{"id":"21805"},"glyph":{"id":"21806"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21807"},"selection_glyph":null,"view":{"id":"21809"}},"id":"21808","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"21744"}},"id":"21738","type":"BoxZoomTool"},{"attributes":{},"id":"21733","type":"BasicTicker"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21813","type":"Dash"},{"attributes":{"data_source":{"id":"21814"},"glyph":{"id":"21813"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21816"}},"id":"21815","type":"GlyphRenderer"},{"attributes":{},"id":"21737","type":"PanTool"},{"attributes":{},"id":"21736","type":"ResetTool"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2148430013731262},"id":"21810","type":"Span"},{"attributes":{},"id":"21834","type":"BasicTickFormatter"},{"attributes":{},"id":"21742","type":"SaveTool"},{"attributes":{},"id":"21739","type":"WheelZoomTool"},{"attributes":{"source":{"id":"21814"}},"id":"21816","type":"CDSView"},{"attributes":{"overlay":{"id":"21745"}},"id":"21740","type":"LassoSelectTool"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"bcJe+Pxh3D88Eu5SUq3NPyL+eQAns8s/mzeOVbYJ1z+YvNPsmzTWP87j+UyletE/sIr1s8Bnzz8YN/msnr7PP8gPs0h4ec8/RNGUEkZ90z8g12riYrDUP+AlTxYjYc4/UBTOvdhAzD+AFwpyJ0DOPxDSzcUXbc4/kDNyJsikyj9AY0p3Si3PP8AWdsF70MQ/4I8dAXxLyD+grhW5nZrSPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21838"},"selection_policy":{"id":"21839"}},"id":"21805","type":"ColumnDataSource"},{"attributes":{},"id":"21756","type":"DataRange1d"},{"attributes":{},"id":"21822","type":"BasicTickFormatter"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.1782444431478369},"id":"21797","type":"Span"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21799","type":"Dash"},{"attributes":{},"id":"21726","type":"LinearScale"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21812","type":"Span"},{"attributes":{},"id":"21838","type":"Selection"},{"attributes":{"children":[[{"id":"21719"},0,0],[{"id":"21755"},0,1]]},"id":"21843","type":"GridBox"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21806","type":"Circle"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21780","type":"BoxAnnotation"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2515582690238702},"id":"21796","type":"Span"},{"attributes":{},"id":"21839","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"21800"},"glyph":{"id":"21799"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21802"}},"id":"21801","type":"GlyphRenderer"},{"attributes":{},"id":"21826","type":"Selection"},{"attributes":{"source":{"id":"21800"}},"id":"21802","type":"CDSView"},{"attributes":{"source":{"id":"21791"}},"id":"21795","type":"CDSView"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21832"},"ticker":{"id":"21765"}},"id":"21764","type":"LinearAxis"},{"attributes":{},"id":"21827","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"21791"},"glyph":{"id":"21792"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21793"},"selection_glyph":null,"view":{"id":"21795"}},"id":"21794","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21798","type":"Span"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21744","type":"BoxAnnotation"},{"attributes":{},"id":"21840","type":"Selection"},{"attributes":{"axis":{"id":"21764"},"ticker":null},"id":"21767","type":"Grid"},{"attributes":{},"id":"21841","type":"UnionRenderers"},{"attributes":{"axis":{"id":"21732"},"dimension":1,"ticker":null},"id":"21735","type":"Grid"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21781","type":"PolyAnnotation"},{"attributes":{},"id":"21828","type":"Selection"},{"attributes":{},"id":"21762","type":"LinearScale"},{"attributes":{},"id":"21829","type":"UnionRenderers"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21745","type":"PolyAnnotation"},{"attributes":{"end":1,"start":-0.05},"id":"21722","type":"DataRange1d"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21772"},{"id":"21773"},{"id":"21774"},{"id":"21775"},{"id":"21776"},{"id":"21777"},{"id":"21778"},{"id":"21779"}]},"id":"21782","type":"Toolbar"},{"attributes":{},"id":"21724","type":"LinearScale"},{"attributes":{"children":[{"id":"21845"},{"id":"21843"}]},"id":"21846","type":"Column"},{"attributes":{"below":[{"id":"21764"}],"center":[{"id":"21767"},{"id":"21771"}],"left":[{"id":"21768"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21808"},{"id":"21810"},{"id":"21811"},{"id":"21812"},{"id":"21815"}],"title":{"id":"21817"},"toolbar":{"id":"21782"},"toolbar_location":null,"x_range":{"id":"21756"},"x_scale":{"id":"21760"},"y_range":{"id":"21758"},"y_scale":{"id":"21762"}},"id":"21755","subtype":"Figure","type":"Plot"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21807","type":"Circle"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21820"},"ticker":{"id":"21729"}},"id":"21728","type":"LinearAxis"},{"attributes":{"toolbars":[{"id":"21746"},{"id":"21782"}],"tools":[{"id":"21736"},{"id":"21737"},{"id":"21738"},{"id":"21739"},{"id":"21740"},{"id":"21741"},{"id":"21742"},{"id":"21743"},{"id":"21772"},{"id":"21773"},{"id":"21774"},{"id":"21775"},{"id":"21776"},{"id":"21777"},{"id":"21778"},{"id":"21779"}]},"id":"21844","type":"ProxyToolbar"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"TOY7mm62yD/upazjteTNP/DqXfy8Lcs/HHCIRHVOzT9orzckCyTOP6y8CkfbP8w/iNx/9NF5yz+IfM+LFJ/NPygi8KloJ8w/oK0q3zNVzD/YcLH58jHPP6hWLqhoMNQ/IGaMl5nu0j8AebTCVJLRP+D0hUiIM80/uNpKBpu90z/A00Fq0J3TP2iCFjW8ldY/kLBgWUSo1j8QCPgZgPnXPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21826"},"selection_policy":{"id":"21827"}},"id":"21791","type":"ColumnDataSource"},{"attributes":{"toolbar":{"id":"21844"},"toolbar_location":"above"},"id":"21845","type":"ToolbarBox"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21736"},{"id":"21737"},{"id":"21738"},{"id":"21739"},{"id":"21740"},{"id":"21741"},{"id":"21742"},{"id":"21743"}]},"id":"21746","type":"Toolbar"},{"attributes":{},"id":"21720","type":"DataRange1d"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21793","type":"Circle"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21792","type":"Circle"},{"attributes":{},"id":"21765","type":"BasicTicker"},{"attributes":{"end":1,"start":-0.05},"id":"21758","type":"DataRange1d"},{"attributes":{},"id":"21729","type":"BasicTicker"},{"attributes":{},"id":"21760","type":"LinearScale"},{"attributes":{"below":[{"id":"21728"}],"center":[{"id":"21731"},{"id":"21735"}],"left":[{"id":"21732"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21794"},{"id":"21796"},{"id":"21797"},{"id":"21798"},{"id":"21801"}],"title":{"id":"21803"},"toolbar":{"id":"21746"},"toolbar_location":null,"x_range":{"id":"21720"},"x_scale":{"id":"21724"},"y_range":{"id":"21722"},"y_scale":{"id":"21726"}},"id":"21719","subtype":"Figure","type":"Plot"}],"root_ids":["21846"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"016603b1-7e9f-425b-b569-618f0bd26624","root_ids":["21846"],"roots":{"21846":"09501135-428c-4582-9e52-32ab60c285b7"}}];
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