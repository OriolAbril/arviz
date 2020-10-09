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
    
      
      
    
      var element = document.getElementById("50f34a4f-2036-4b3c-b98f-d895e056fc76");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '50f34a4f-2036-4b3c-b98f-d895e056fc76' but no matching script tag was found.")
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
                    
                  var docs_json = '{"977eb821-13f6-4ee3-9662-0aa0a9bfb30c":{"roots":{"references":[{"attributes":{"toolbars":[{"id":"85713"},{"id":"85749"}],"tools":[{"id":"85703"},{"id":"85704"},{"id":"85705"},{"id":"85706"},{"id":"85707"},{"id":"85708"},{"id":"85709"},{"id":"85710"},{"id":"85739"},{"id":"85740"},{"id":"85741"},{"id":"85742"},{"id":"85743"},{"id":"85744"},{"id":"85745"},{"id":"85746"}]},"id":"85811","type":"ProxyToolbar"},{"attributes":{"data":{"rug_x":{"__ndarray__":"fV36E1z/6j89DycBWWfXP73VlTJ7YsE/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/Oqay45Jr6D86SHRMZcflP7HThSU1z+I/VdSaCTtd6D9Hvab2ZmSwP8UA5kQ6d4M/0GULqag1oz+aI/Yi4T7rP55DEK8H/NA/qgGd6qjX1D+VqnS/h2ThP/UnuP7VN+Q/TGXHJdeQ2z8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"85805"},"selection_policy":{"id":"85806"}},"id":"85781","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"85695"}],"center":[{"id":"85698"},{"id":"85702"}],"left":[{"id":"85699"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"85761"},{"id":"85763"},{"id":"85764"},{"id":"85765"},{"id":"85768"}],"title":{"id":"85770"},"toolbar":{"id":"85713"},"toolbar_location":null,"x_range":{"id":"85687"},"x_scale":{"id":"85691"},"y_range":{"id":"85689"},"y_scale":{"id":"85693"}},"id":"85686","subtype":"Figure","type":"Plot"},{"attributes":{"source":{"id":"85781"}},"id":"85783","type":"CDSView"},{"attributes":{"source":{"id":"85772"}},"id":"85776","type":"CDSView"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"85780","type":"Dash"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2148430013731262},"id":"85777","type":"Span"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"85799"},"ticker":{"id":"85732"}},"id":"85731","type":"LinearAxis"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.15209716424958658},"id":"85778","type":"Span"},{"attributes":{"below":[{"id":"85731"}],"center":[{"id":"85734"},{"id":"85738"}],"left":[{"id":"85735"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"85775"},{"id":"85777"},{"id":"85778"},{"id":"85779"},{"id":"85782"}],"title":{"id":"85784"},"toolbar":{"id":"85749"},"toolbar_location":null,"x_range":{"id":"85723"},"x_scale":{"id":"85727"},"y_range":{"id":"85725"},"y_scale":{"id":"85729"}},"id":"85722","subtype":"Figure","type":"Plot"},{"attributes":{"end":1,"start":-0.05},"id":"85725","type":"DataRange1d"},{"attributes":{"data_source":{"id":"85781"},"glyph":{"id":"85780"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"85783"}},"id":"85782","type":"GlyphRenderer"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"85801"},"ticker":{"id":"85736"}},"id":"85735","type":"LinearAxis"},{"attributes":{"text":"mu"},"id":"85784","type":"Title"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"85789"},"ticker":{"id":"85700"}},"id":"85699","type":"LinearAxis"},{"attributes":{},"id":"85727","type":"LinearScale"},{"attributes":{},"id":"85729","type":"LinearScale"},{"attributes":{"callback":null},"id":"85746","type":"HoverTool"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"85760","type":"Circle"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"85779","type":"Span"},{"attributes":{},"id":"85687","type":"DataRange1d"},{"attributes":{},"id":"85732","type":"BasicTicker"},{"attributes":{"axis":{"id":"85731"},"ticker":null},"id":"85734","type":"Grid"},{"attributes":{},"id":"85691","type":"LinearScale"},{"attributes":{"toolbar":{"id":"85811"},"toolbar_location":"above"},"id":"85812","type":"ToolbarBox"},{"attributes":{"callback":null},"id":"85710","type":"HoverTool"},{"attributes":{"end":1,"start":-0.05},"id":"85689","type":"DataRange1d"},{"attributes":{"axis":{"id":"85735"},"dimension":1,"ticker":null},"id":"85738","type":"Grid"},{"attributes":{},"id":"85736","type":"BasicTicker"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"85739"},{"id":"85740"},{"id":"85741"},{"id":"85742"},{"id":"85743"},{"id":"85744"},{"id":"85745"},{"id":"85746"}]},"id":"85749","type":"Toolbar"},{"attributes":{},"id":"85700","type":"BasicTicker"},{"attributes":{"overlay":{"id":"85747"}},"id":"85741","type":"BoxZoomTool"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"85787"},"ticker":{"id":"85696"}},"id":"85695","type":"LinearAxis"},{"attributes":{},"id":"85740","type":"PanTool"},{"attributes":{},"id":"85693","type":"LinearScale"},{"attributes":{},"id":"85739","type":"ResetTool"},{"attributes":{},"id":"85696","type":"BasicTicker"},{"attributes":{},"id":"85703","type":"ResetTool"},{"attributes":{},"id":"85745","type":"SaveTool"},{"attributes":{},"id":"85742","type":"WheelZoomTool"},{"attributes":{"axis":{"id":"85695"},"ticker":null},"id":"85698","type":"Grid"},{"attributes":{"overlay":{"id":"85748"}},"id":"85743","type":"LassoSelectTool"},{"attributes":{},"id":"85744","type":"UndoTool"},{"attributes":{"axis":{"id":"85699"},"dimension":1,"ticker":null},"id":"85702","type":"Grid"},{"attributes":{},"id":"85723","type":"DataRange1d"},{"attributes":{"overlay":{"id":"85711"}},"id":"85705","type":"BoxZoomTool"},{"attributes":{},"id":"85704","type":"PanTool"},{"attributes":{},"id":"85706","type":"WheelZoomTool"},{"attributes":{},"id":"85804","type":"UnionRenderers"},{"attributes":{},"id":"85709","type":"SaveTool"},{"attributes":{"overlay":{"id":"85712"}},"id":"85707","type":"LassoSelectTool"},{"attributes":{},"id":"85708","type":"UndoTool"},{"attributes":{},"id":"85803","type":"Selection"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"85703"},{"id":"85704"},{"id":"85705"},{"id":"85706"},{"id":"85707"},{"id":"85708"},{"id":"85709"},{"id":"85710"}]},"id":"85713","type":"Toolbar"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"85748","type":"PolyAnnotation"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"85747","type":"BoxAnnotation"},{"attributes":{},"id":"85799","type":"BasicTickFormatter"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"85759","type":"Circle"},{"attributes":{},"id":"85801","type":"BasicTickFormatter"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"TOY7mm62yD/upazjteTNP/DqXfy8Lcs/HHCIRHVOzT9orzckCyTOP6y8CkfbP8w/iNx/9NF5yz+IfM+LFJ/NPygi8KloJ8w/oK0q3zNVzD/YcLH58jHPP6hWLqhoMNQ/IGaMl5nu0j8AebTCVJLRP+D0hUiIM80/uNpKBpu90z/A00Fq0J3TP2iCFjW8ldY/kLBgWUSo1j8QCPgZgPnXPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"85791"},"selection_policy":{"id":"85792"}},"id":"85758","type":"ColumnDataSource"},{"attributes":{},"id":"85787","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"85711","type":"BoxAnnotation"},{"attributes":{},"id":"85805","type":"Selection"},{"attributes":{},"id":"85789","type":"BasicTickFormatter"},{"attributes":{},"id":"85806","type":"UnionRenderers"},{"attributes":{},"id":"85791","type":"Selection"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"85712","type":"PolyAnnotation"},{"attributes":{},"id":"85792","type":"UnionRenderers"},{"attributes":{"children":[[{"id":"85686"},0,0],[{"id":"85722"},0,1]]},"id":"85810","type":"GridBox"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"bcJe+Pxh3D88Eu5SUq3NPyL+eQAns8s/mzeOVbYJ1z+YvNPsmzTWP87j+UyletE/sIr1s8Bnzz8YN/msnr7PP8gPs0h4ec8/RNGUEkZ90z8g12riYrDUP+AlTxYjYc4/UBTOvdhAzD+AFwpyJ0DOPxDSzcUXbc4/kDNyJsikyj9AY0p3Si3PP8AWdsF70MQ/4I8dAXxLyD+grhW5nZrSPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"85803"},"selection_policy":{"id":"85804"}},"id":"85772","type":"ColumnDataSource"},{"attributes":{},"id":"85793","type":"Selection"},{"attributes":{"data":{"rug_x":{"__ndarray__":"jQwCEA1Gsz9bBMLb9PjIP1Qd9Zram7E/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/aF85wG2piz9oXznAbamLP2hfOcBtqYs/aF85wG2piz9oXznAbamLP2hfOcBtqYs/pI3yRkqEyT9c/+ob+nG6P4OiBeyjALU/mx4fY+a33D9wF2c1cbTRPy51Aws2htg/P07TEgOYwz9FOncTGHy5P/BQ+ANPucc/uRO6PYJJzj9CRCWTDYpmP+UvVnGFsrI/8JR71fNwcj8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"85793"},"selection_policy":{"id":"85794"}},"id":"85767","type":"ColumnDataSource"},{"attributes":{},"id":"85794","type":"UnionRenderers"},{"attributes":{"source":{"id":"85767"}},"id":"85769","type":"CDSView"},{"attributes":{"data_source":{"id":"85758"},"glyph":{"id":"85759"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"85760"},"selection_glyph":null,"view":{"id":"85762"}},"id":"85761","type":"GlyphRenderer"},{"attributes":{"source":{"id":"85758"}},"id":"85762","type":"CDSView"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"85766","type":"Dash"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2515582690238702},"id":"85763","type":"Span"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.1782444431478369},"id":"85764","type":"Span"},{"attributes":{"data_source":{"id":"85767"},"glyph":{"id":"85766"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"85769"}},"id":"85768","type":"GlyphRenderer"},{"attributes":{"text":"tau"},"id":"85770","type":"Title"},{"attributes":{"data_source":{"id":"85772"},"glyph":{"id":"85773"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"85774"},"selection_glyph":null,"view":{"id":"85776"}},"id":"85775","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"85765","type":"Span"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"85774","type":"Circle"},{"attributes":{"children":[{"id":"85812"},{"id":"85810"}]},"id":"85813","type":"Column"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"85773","type":"Circle"}],"root_ids":["85813"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"977eb821-13f6-4ee3-9662-0aa0a9bfb30c","root_ids":["85813"],"roots":{"85813":"50f34a4f-2036-4b3c-b98f-d895e056fc76"}}];
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